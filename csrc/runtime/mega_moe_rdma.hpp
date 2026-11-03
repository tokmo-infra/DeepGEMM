#pragma once

#include "jit.hpp"
#include "../utils/torch_compat.hpp"
#include "../utils/exception.hpp"
#include <deep_gemm/layout/mega_moe_rdma.cuh>
#include <deep_gemm/layout/mega_moe.cuh>

#ifdef DG_ENABLE_NVSHMEM
#include <nvshmem.h>
#include <nvshmemx.h>
#include <unordered_map>
#endif

namespace deep_gemm::rdma {

inline int64_t abi_version() {
#ifdef DG_ENABLE_NVSHMEM
    return 1;
#else
    return 0;
#endif
}

#ifdef DG_ENABLE_NVSHMEM
inline void check(int status) { DG_HOST_ASSERT(status == 0); }

// Process-scoped resources deliberately outlive all tensor views and captured
// graphs. Never run collective free/finalize from a Python/C++ destructor.
struct State {
    bool initialized = false;
    int device = -1;
    std::vector<void*> allocations;
    std::unordered_map<void*, size_t> sizes;
    std::unordered_map<CUlibrary, std::shared_ptr<deep_jit::cuda::Kernel>> kernels;
};
inline State& state() { static auto* value = new State(); return *value; }

inline std::vector<int64_t> unique_id() {
    nvshmemx_uniqueid_t uid = NVSHMEMX_UNIQUEID_INITIALIZER;
    check(nvshmemx_get_uniqueid(&uid));
    const auto* bytes = reinterpret_cast<const uint8_t*>(&uid);
    return {bytes, bytes + sizeof(uid)};
}

inline void init(const std::vector<int64_t>& bytes, int64_t rank, int64_t size) {
    DG_HOST_ASSERT(not state().initialized);
    DG_HOST_ASSERT(nvshmemx_init_status() == NVSHMEM_STATUS_NOT_INITIALIZED);
    DG_HOST_ASSERT(rank >= 0 and rank < size and size <= layout::kNumMaxRanks);
    nvshmemx_uniqueid_t uid = NVSHMEMX_UNIQUEID_INITIALIZER;
    DG_HOST_ASSERT(bytes.size() == sizeof(uid));
    auto* dst = reinterpret_cast<uint8_t*>(&uid);
    for (size_t i = 0; i < bytes.size(); ++i) {
        DG_HOST_ASSERT(bytes[i] >= 0 and bytes[i] < 256);
        dst[i] = bytes[i];
    }
    check(cudaGetDevice(&state().device));
    nvshmemx_init_attr_t attr = NVSHMEMX_INIT_ATTR_INITIALIZER;
    nvshmemx_set_attr_uniqueid_args(rank, size, &uid, &attr);
    check(nvshmemx_hostlib_init_attr(NVSHMEMX_INIT_WITH_UNIQUEID, &attr));
    DG_HOST_ASSERT(nvshmem_my_pe() == rank and nvshmem_n_pes() == size);
    state().initialized = true;
}

inline std::tuple<torch::stable::Tensor, std::vector<int64_t>> allocate(int64_t bytes) {
    DG_HOST_ASSERT(state().initialized and bytes > 0);
    int device;
    check(cudaGetDevice(&device));
    DG_HOST_ASSERT(device == state().device);
    auto* ptr = nvshmem_align(128, bytes);
    DG_HOST_ASSERT(ptr != nullptr);
    state().allocations.push_back(ptr);
    state().sizes.emplace(ptr, bytes);
    check(cudaMemset(ptr, 0, bytes));
    check(cudaDeviceSynchronize());
    nvshmem_barrier_all();
    std::vector<int64_t> peers;
    for (int pe = 0; pe < nvshmem_n_pes(); ++pe)
        peers.push_back(reinterpret_cast<int64_t>(nvshmem_ptr(ptr, pe)));
    return {torch_compat::from_blob(ptr, {bytes}, torch::stable::Device(torch::headeronly::kCUDA, state().device),
                                  torch::headeronly::ScalarType::Char), peers};
}

inline void validate(const torch::stable::Tensor& main, const torch::stable::Tensor& scratch,
                     const std::vector<int64_t>& peers, int rank, uint64_t required) {
    DG_HOST_ASSERT(state().initialized);
    DG_HOST_ASSERT(rank == nvshmem_my_pe() and peers.size() == static_cast<size_t>(nvshmem_n_pes()));
    DG_HOST_ASSERT(main.get_device_index() == state().device and scratch.get_device_index() == state().device);
    DG_HOST_ASSERT(state().sizes.contains(main.mutable_data_ptr()));
    DG_HOST_ASSERT(state().sizes.contains(scratch.mutable_data_ptr()));
    DG_HOST_ASSERT(state().sizes.at(scratch.mutable_data_ptr()) >= required);
    for (int pe = 0; pe < nvshmem_n_pes(); ++pe)
        DG_HOST_ASSERT(peers[pe] == reinterpret_cast<int64_t>(nvshmem_ptr(main.mutable_data_ptr(), pe)));
}

inline std::string shell_quote(const std::string& value) {
    std::string result = "'";
    for (char c : value) result += c == '\'' ? "'\\''" : std::string(1, c);
    return result + "'";
}

// DeepJIT's native compiler emits a single cubin. NVSHMEM needs an explicit
// relocatable object + device link step; keep its cache/loader unchanged.
struct Backend : deep_jit::CUDA {
    using deep_jit::CUDA::CUDA;
    void compile(const std::string& source, std::filesystem::path dir,
                 const deep_jit::Env& env, const deep_jit::Config& config,
                 const CompilerOptions& options) const {
        dir = std::filesystem::absolute(dir);
        deep_jit::write_file_sync(dir / "kernel.cu", source);
        std::string commands;
        auto run = [&](const std::vector<std::string>& args) {
            std::string command;
            for (const auto& arg : args) command += shell_quote(arg) + " ";
            commands += command + "\n";
            return deep_jit::call_external_command(command, env.get<bool>("JIT_PRINT_COMPILER_COMMAND", false));
        };
        std::vector<std::string> args{toolkit.nvcc.string(), "-dc", (dir / "kernel.cu").string(),
                                      "-o", (dir / "kernel.o").string()};
        auto flags = options.get_flags();
        // With -dc, -arch=sm_103f also emits generic compute_103 PTX, which
        // cannot assemble MegaMoE's family-specific tcgen05 instructions.
        const auto target = "--generate-code=arch=compute_" + *options.arch + ",code=sm_" + *options.arch;
        flags.front() = target;
        args.insert(args.end(), flags.begin(), flags.end());
        for (const auto& path : config.include_dirs) {
            args.emplace_back("-I"); args.push_back(path.string());
        }
        const auto compile_log = run(args);
        const auto link_log = run({toolkit.nvcc.string(), "-dlink", "--cubin",
            target, (dir / "kernel.o").string(), DG_NVSHMEM_DEVICE_LIB,
            "-o", (dir / "kernel.cubin").string()});
        deep_jit::write_file_sync(dir / "build.log", commands + compile_log + link_log);
        deep_jit::write_file_sync(dir / "meta.json", deep_jit::json(deep_jit::json::object_t{
            {"config", config.to_json()}, {"compiler_info", compiler_info.to_json()},
            {"compiler_options", options.to_json()}, {"commands", commands}
        }).dump());
        DG_HOST_ASSERT(not options.check_no_spills.value_or(false) or
            not std::regex_search(compile_log + link_log, std::regex(R"(spilled\s+to\s+local\s+memory)", std::regex::icase)));
        DG_HOST_ASSERT(std::filesystem::file_size(dir / "kernel.cubin") > 0);
    }
};

inline std::shared_ptr<deep_jit::cuda::Kernel> compile(const std::string& source) {
    static auto* runtime = [] {
        auto config = jit->config;
        config.include_dirs.emplace_back(DG_NVSHMEM_INCLUDE_DIR);
        config.extra_signature += "-megamoe-rdma-v2-" + deep_jit::hash::get_hex_digest(
            deep_jit::read(DG_NVSHMEM_DEVICE_LIB));
        return new deep_jit::Runtime<Backend>(config);
    }();
    auto options = jit->default_compiler_options;
    options.extra_nvcc_flags.emplace_back("-DDG_MEGAMOE_NVSHMEM=1");
    const auto kernel = runtime->compile("sm100_fp8_fp4_mega_moe_rdma", source, options);
    if (not state().kernels.contains(kernel->library_handle)) {
        cudaStreamCaptureStatus capture;
        check(cudaStreamIsCapturing(reinterpret_cast<cudaStream_t>(
            deep_jit::cuda::get_current_cuda_stream(state().device)), &capture));
        DG_HOST_ASSERT(capture == cudaStreamCaptureStatusNone);
        CUmodule module;
        DG_HOST_ASSERT(cuLibraryGetModule(&module, kernel->library_handle) == CUDA_SUCCESS);
        check(nvshmemx_cumodule_init(module));
        state().kernels.emplace(kernel->library_handle, kernel);
    }
    return kernel;
}

template <typename... Args>
inline void launch(const std::shared_ptr<deep_jit::cuda::Kernel>& kernel,
                   deep_jit::cuda::LaunchOptions options, const Args&... args) {
    options = jit->default_launch_options.override_with(options);
    DG_HOST_ASSERT(not options.enable_pdl.value_or(false));
    CUkernel handle;
    DG_HOST_ASSERT(cuLibraryEnumerateKernels(&handle, 1, kernel->library_handle) == CUDA_SUCCESS);
    const void* func = reinterpret_cast<const void*>(handle); // CUDA >=12.8 kernel-handle interop
    check(cudaFuncSetAttribute(func, cudaFuncAttributeMaxDynamicSharedMemorySize, *options.num_smem_bytes));
    cudaLaunchAttribute attributes[2]{};
    attributes[0].id = cudaLaunchAttributeClusterDimension;
    attributes[0].val.clusterDim = {options.cluster_dim->x, 1, 1};
    attributes[1].id = cudaLaunchAttributeCooperative;
    attributes[1].val.cooperative = 1;
    nvshmemx_collective_launch_attr_t launch{};
    launch.cuda_config.gridDim = *options.grid_dim;
    launch.cuda_config.blockDim = *options.block_dim;
    launch.cuda_config.dynamicSmemBytes = *options.num_smem_bytes;
    launch.cuda_config.stream = reinterpret_cast<cudaStream_t>(
        options.stream.value_or(deep_jit::cuda::get_current_cuda_stream(state().device)));
    launch.cuda_config.attrs = attributes;
    launch.cuda_config.numAttrs = 2;
    int active_clusters = 0;
    check(cudaOccupancyMaxActiveClusters(&active_clusters, func, &launch.cuda_config));
    DG_HOST_ASSERT(options.grid_dim->x <= active_clusters * options.cluster_dim->x);
    void* pointers[]{deep_jit::cuda::kernel_arg_pointer(args)...};
    check(nvshmemx_collective_launch_attr(&launch, func, pointers));
}
#endif

inline std::shared_ptr<deep_jit::cuda::Kernel> compile(const std::string& source, bool enabled) {
    if (not enabled) return jit->compile("sm100_fp8_fp4_mega_moe", source);
#ifdef DG_ENABLE_NVSHMEM
    return compile(source);
#else
    DG_HOST_UNREACHABLE("Build DeepGEMM with DG_ENABLE_NVSHMEM=1 for MegaMoE RDMA");
#endif
}

template <typename... Args>
inline void launch(bool enabled, const std::shared_ptr<deep_jit::cuda::Kernel>& kernel,
                   const deep_jit::cuda::LaunchOptions& options, const Args&... args) {
    if (not enabled) { jit->launch(kernel, options, args...); return; }
#ifdef DG_ENABLE_NVSHMEM
    launch(kernel, options, args...);
#else
    DG_HOST_UNREACHABLE("MegaMoE RDMA is not built");
#endif
}

} // namespace deep_gemm::rdma

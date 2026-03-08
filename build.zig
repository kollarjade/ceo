const std = @import("std");
const builtin = @import("builtin");

const Build = std.Build;
const Step = Build.Step;
const Module = Build.Module;
const Compile = Build.Step.Compile;

pub fn build(b: *Build) !void {
    const target = b.standardTargetOptions(.{});
    const optimize = b.standardOptimizeOption(.{});

    // Configuration options
    const cuda_arch = b.option([]const u8, "cuda-arch", "CUDA architecture (e.g., sm_100)") orelse "sm_100";
    const cuda_path = b.option([]const u8, "cuda-path", "Path to CUDA installation") orelse "/usr/local/cuda";
    const enable_nccl = b.option(bool, "enable-nccl", "Enable NCCL support") orelse true;
    const enable_futhark = b.option(bool, "enable-futhark", "Enable Futhark kernel support") orelse false;
    const enable_fp8 = b.option(bool, "enable-fp8", "Enable FP8 training support") orelse true;
    const enable_profiling = b.option(bool, "enable-profiling", "Enable profiling hooks") orelse false;

    // Create the main executable
    const exe = b.addExecutable(.{
        .name = "efla-train",
        .root_module = b.createModule(.{
            .root_source_file = b.path("src/main.zig"),
            .target = target,
            .optimize = optimize,
        }),
    });

    // Add C include paths for CUDA
    exe.addIncludePath(b.path("kernels/cuda/include"));
    exe.addIncludePath(.{ .cwd_relative = b.fmt("{s}/include", .{cuda_path}) });

    // Link CUDA libraries
    exe.linkLibC();
    exe.linkSystemLibrary("cuda");
    exe.linkSystemLibrary("cudart");
    exe.linkSystemLibrary("cublas");
    exe.linkSystemLibrary("cublasLt");
    exe.linkSystemLibrary("cusparse");
    exe.linkSystemLibrary("cusolver");

    if (enable_nccl) {
        exe.linkSystemLibrary("nccl");
        exe.root_module.addCMacro("ENABLE_NCCL", "1");
    }

    if (enable_fp8) {
        exe.root_module.addCMacro("ENABLE_FP8", "1");
    }

    if (enable_profiling) {
        exe.root_module.addCMacro("ENABLE_PROFILING", "1");
        exe.linkSystemLibrary("cupti");
    }

    // Add library search paths
    exe.addLibraryPath(.{ .cwd_relative = b.fmt("{s}/lib64", .{cuda_path}) });
    exe.addLibraryPath(.{ .cwd_relative = b.fmt("{s}/lib", .{cuda_path}) });

    // Build CUDA kernels
    const cuda_kernels = try buildCudaKernels(b, exe, cuda_arch, cuda_path, optimize);

    // Link CUDA kernel objects
    for (cuda_kernels) |obj| {
        exe.linkLibrary(obj);
    }

    // Build Futhark kernels if enabled
    if (enable_futhark) {
        try buildFutharkKernels(b, exe, optimize);
        exe.root_module.addCMacro("ENABLE_FUTHARK", "1");
    }

    // Install the executable
    b.installArtifact(exe);

    // Create run step
    const run_cmd = b.addRunArtifact(exe);
    run_cmd.step.dependOn(b.getInstallStep());

    if (b.args) |args| {
        run_cmd.addArgs(args);
    }

    const run_step = b.step("run", "Run the training system");
    run_step.dependOn(&run_cmd.step);

    // Create test step
    const unit_tests = b.addTest(.{
        .root_module = b.createModule(.{
            .root_source_file = b.path("tests/unit/main.zig"),
            .target = target,
            .optimize = optimize,
        }),
    });

    unit_tests.addIncludePath(b.path("kernels/cuda/include"));
    unit_tests.linkLibC();

    const run_unit_tests = b.addRunArtifact(unit_tests);
    const test_step = b.step("test", "Run unit tests");
    test_step.dependOn(&run_unit_tests.step);

    // Kernel compilation step
    const kernels_step = b.step("kernels", "Build only CUDA kernels");
    for (cuda_kernels) |obj| {
        kernels_step.dependOn(&obj.step);
    }

    // Smoke test step
    const smoke_step = b.step("smoke", "Run smoke test");
    const smoke_run = b.addRunArtifact(exe);
    smoke_run.addArgs(&[_][]const u8{ "smoke-test", "--config", "configs/smoke.yaml" });
    smoke_step.dependOn(&smoke_run.step);

    // Clean step
    const clean_step = b.step("clean", "Clean build artifacts");
    clean_step.makeFn = cleanBuild;
    clean_step.dependOn(&b.addRemoveDirTree(b.path("zig-out")).step);

    // Docs generation step
    const docs_step = b.step("docs", "Generate documentation");
    const docs_obj = b.addTest(.{
        .root_module = b.createModule(.{
            .root_source_file = b.path("src/main.zig"),
            .target = target,
            .optimize = optimize,
        }),
    });
    docs_obj.emit_docs = .emit;
    const docs_install = b.addInstallDirectory(.{
        .source_dir = docs_obj.getEmittedDocs(),
        .install_dir = .prefix,
        .install_subdir = "docs",
    });
    docs_step.dependOn(&docs_install.step);
}

fn buildCudaKernels(b: *Build, exe: *Compile, arch: []const u8, cuda_path: []const u8, optimize: std.builtin.OptimizeMode) ![]*Compile {
    var kernels = std.ArrayList(*Compile).init(b.allocator);

    const cuda_files = [_][]const u8{
        "kernels/cuda/gemm.cu",
        "kernels/cuda/layernorm.cu",
        "kernels/cuda/rmsnorm.cu",
        "kernels/cuda/gelu.cu",
        "kernels/cuda/softmax.cu",
        "kernels/cuda/attention.cu",
        "kernels/cuda/efla.cu",
        "kernels/cuda/prism.cu",
        "kernels/cuda/shortconv.cu",
        "kernels/cuda/scan.cu",
        "kernels/cuda/embedding.cu",
        "kernels/cuda/cross_entropy.cu",
        "kernels/cuda/scatter.cu",
        "kernels/cuda/gather.cu",
        "kernels/cuda/reduction.cu",
        "kernels/cuda/fp8_ops.cu",
        "kernels/cuda/memory.cu",
        "kernels/cuda/init.cu",
    };

    const nvcc_flags = [_][]const u8{
        "-arch", arch,
        "-O3",
        "--use_fast_math",
        "-DCUDA_ARCH_SM100",
        "--ptxas-options=-v",
        "-lineinfo",
        "--expt-relaxed-constexpr",
        "--expt-extended-lambda",
        "-std=c++20",
    };

    const debug_flags = [_][]const u8{
        "-arch", arch,
        "-O0",
        "-g",
        "-G",
        "-DCUDA_ARCH_SM100",
        "--expt-relaxed-constexpr",
        "--expt-extended-lambda",
        "-std=c++20",
    };

    const flags = if (optimize == .Debug) &debug_flags else &nvcc_flags;

    for (cuda_files) |cuda_file| {
        const name = std.fs.path.stem(cuda_file);
        const lib = b.addSharedLibrary(.{
            .name = b.fmt("cuda_{s}", .{name}),
            .target = exe.root_module.resolved_target.?,
            .optimize = optimize,
        });

        lib.addCSourceFiles(.{
            .files = &.{cuda_file},
            .flags = flags,
        });

        lib.addIncludePath(b.path("kernels/cuda/include"));
        lib.addIncludePath(.{ .cwd_relative = b.fmt("{s}/include", .{cuda_path}) });
        lib.linkLibC();
        lib.linkSystemLibrary("cudart");
        lib.linkSystemLibrary("cublas");
        lib.linkSystemLibrary("cublasLt");

        lib.addLibraryPath(.{ .cwd_relative = b.fmt("{s}/lib64", .{cuda_path}) });
        lib.addLibraryPath(.{ .cwd_relative = b.fmt("{s}/lib", .{cuda_path}) });

        try kernels.append(lib);
    }

    return kernels.toOwnedSlice();
}

fn buildFutharkKernels(b: *Build, exe: *Compile, optimize: std.builtin.OptimizeMode) !void {
    _ = b;
    _ = exe;
    _ = optimize;
    // Futhark kernels would be built here using futhark cuda
    // For now, this is a placeholder that would invoke:
    // futhark cuda kernels/futhark/*.fut -o kernels/futhark/generated/
}

fn cleanBuild(step: *Step) !void {
    _ = step;
    // Additional cleanup logic
}

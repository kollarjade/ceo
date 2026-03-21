const std = @import("std");
const builtin = @import("builtin");

const Build = std.Build;
const Module = Build.Module;
const Compile = Build.Step.Compile;

const CudaArtifacts = struct {
    step: *Build.Step,
    library_names: [][]const u8,
};

pub fn build(b: *Build) !void {
    _ = builtin;
    _ = Module;

    const target = b.standardTargetOptions(.{});
    const optimize = b.standardOptimizeOption(.{});

    const cuda_arch = b.option([]const u8, "cuda-arch", "CUDA architecture (e.g., sm_100)") orelse "sm_100";
    const cuda_path = b.option([]const u8, "cuda-path", "Path to CUDA installation") orelse "/usr/local/cuda";
    const enable_nccl = b.option(bool, "enable-nccl", "Enable NCCL support") orelse true;
    const enable_futhark = b.option(bool, "enable-futhark", "Enable Futhark kernel support") orelse false;
    const enable_fp8 = b.option(bool, "enable-fp8", "Enable FP8 training support") orelse true;
    const enable_profiling = b.option(bool, "enable-profiling", "Enable profiling hooks") orelse false;

    const exe = b.addExecutable(.{
        .name = "efla-train",
        .root_module = b.createModule(.{
            .root_source_file = b.path("src/main.zig"),
            .target = target,
            .optimize = optimize,
        }),
    });

    exe.addIncludePath(b.path("kernels/cuda/include"));
    exe.addIncludePath(.{ .cwd_relative = b.fmt("{s}/include", .{cuda_path}) });
    exe.addLibraryPath(.{ .cwd_relative = b.fmt("{s}/lib64", .{cuda_path}) });
    exe.addLibraryPath(.{ .cwd_relative = b.fmt("{s}/lib", .{cuda_path}) });
    exe.addLibraryPath(.{ .cwd_relative = "build/cuda" });

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

    const cuda_artifacts = try buildCudaKernels(b, cuda_arch, cuda_path);
    exe.step.dependOn(cuda_artifacts.step);

    for (cuda_artifacts.library_names) |lib_name| {
        exe.linkSystemLibrary(lib_name);
    }

    if (enable_futhark) {
        try buildFutharkKernels(b, exe, optimize);
        exe.root_module.addCMacro("ENABLE_FUTHARK", "1");
    }

    b.installArtifact(exe);

    const install_cuda_dir = b.addInstallDirectory(.{
        .source_dir = .{ .cwd_relative = "build/cuda" },
        .install_dir = .prefix,
        .install_subdir = "lib",
    });
    install_cuda_dir.step.dependOn(cuda_artifacts.step);
    b.getInstallStep().dependOn(&install_cuda_dir.step);

    const run_cmd = b.addRunArtifact(exe);
    run_cmd.step.dependOn(b.getInstallStep());
    if (b.args) |args| {
        run_cmd.addArgs(args);
    }

    const run_step = b.step("run", "Run the training system");
    run_step.dependOn(&run_cmd.step);

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

    const kernels_step = b.step("kernels", "Build only CUDA kernels");
    kernels_step.dependOn(cuda_artifacts.step);

    const smoke_step = b.step("smoke", "Run smoke test");
    const smoke_run = b.addRunArtifact(exe);
    smoke_run.addArgs(&[_][]const u8{ "smoke-test", "--config", "configs/smoke.yaml" });
    smoke_step.dependOn(&smoke_run.step);

    const clean_step = b.step("clean", "Clean build artifacts");
    clean_step.dependOn(&b.addRemoveDirTree(.{ .cwd_relative = "zig-out" }).step);
    clean_step.dependOn(&b.addRemoveDirTree(.{ .cwd_relative = "zig-cache" }).step);
    clean_step.dependOn(&b.addRemoveDirTree(.{ .cwd_relative = "build/cuda" }).step);

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

fn buildCudaKernels(b: *Build, arch: []const u8, cuda_path: []const u8) !CudaArtifacts {
    const command = b.addSystemCommand(&.{ "bash", "scripts/build_kernels.sh" });
    command.setEnvironmentVariable("CUDA_ARCH", arch);
    command.setEnvironmentVariable("CUDA_PATH", cuda_path);

    const library_names = try collectCudaLibraryNames(b);
    return .{
        .step = &command.step,
        .library_names = library_names,
    };
}

fn buildFutharkKernels(b: *Build, exe: *Compile, optimize: std.builtin.OptimizeMode) !void {
    _ = optimize;
    var dir = std.fs.cwd().openDir("kernels/futhark", .{ .iterate = true }) catch return;
    defer dir.close();

    var iterator = dir.iterate();
    while (try iterator.next()) |entry| {
        if (entry.kind != .file) continue;
        if (!std.mem.endsWith(u8, entry.name, ".c")) continue;

        exe.addCSourceFile(.{
            .file = .{ .cwd_relative = b.fmt("kernels/futhark/{s}", .{entry.name}) },
            .flags = &[_][]const u8{ "-O3" },
        });
    }
}

fn collectCudaLibraryNames(b: *Build) ![][]const u8 {
    var dir = try std.fs.cwd().openDir("kernels/cuda", .{ .iterate = true });
    defer dir.close();

    var names = std.ArrayList([]const u8).init(b.allocator);
    var iterator = dir.iterate();

    while (try iterator.next()) |entry| {
        if (entry.kind != .file) continue;
        if (!std.mem.endsWith(u8, entry.name, ".cu")) continue;
        const stem = std.fs.path.stem(entry.name);
        try names.append(try std.fmt.allocPrint(b.allocator, "cuda_{s}", .{stem}));
    }

    if (names.items.len == 0) {
        return error.NoCudaSources;
    }

    return names.toOwnedSlice();
}

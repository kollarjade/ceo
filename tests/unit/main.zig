const std = @import("std");

// Import all modules for testing
comptime {
    _ = @import("../../src/tensor/dtype.zig");
    _ = @import("../../src/tensor/tensor.zig");
    _ = @import("../../src/tensor/layout.zig");
    _ = @import("../../src/tensor/sharding.zig");
    _ = @import("../../src/util/config.zig");
    _ = @import("../../src/util/yaml.zig");
    _ = @import("../../src/util/rng.zig");
    _ = @import("../../src/util/cli.zig");
    _ = @import("../../src/util/error.zig");
    _ = @import("../../src/util/hash.zig");
    _ = @import("../../src/nn/layers.zig");
    _ = @import("../../src/optim/optimizer.zig");
    _ = @import("../../src/model/efla.zig");
    _ = @import("../../src/model/prism.zig");
    _ = @import("../../src/model/model.zig");
    _ = @import("../../src/data/tokenizer.zig");
    _ = @import("../../src/data/dataset.zig");
    _ = @import("../../src/checkpoint/manager.zig");
    _ = @import("../../src/kernels/efla_kernels.zig");
    _ = @import("../../src/kernels/prism_kernels.zig");
    _ = @import("../../src/kernels/nn_kernels.zig");
    _ = @import("../../src/kernels/optim_kernels.zig");
}

pub fn main() !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();

    const allocator = gpa.allocator();

    std.log.info("Running EFLA Trainer unit tests...", .{});

    // Run all tests
    const result = std.testing.runTests(
        .{
            .allocator = allocator,
            .filter = null,
        },
        @import("build_options").tests,
    );

    if (result.failure_count > 0) {
        std.log.err("Tests failed: {} passed, {} failed", .{ result.success_count, result.failure_count });
        return error.TestFailed;
    }

    std.log.info("All tests passed: {} tests", .{result.success_count});
}

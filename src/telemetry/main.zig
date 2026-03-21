const std = @import("std");
const config_mod = @import("../util/config.zig");

pub const TelemetryConfig = config_mod.TelemetryConfig;

pub const StepMetrics = struct {
    step: usize,
    tokens: usize,
    loss: f32,
    lr: f32,
    grad_norm: f32,
    throughput: f64,
    memory_used: usize,
    memory_total: usize,
    timestamp: i64,
};

pub const Telemetry = struct {
    config: TelemetryConfig,
    rank: usize,
    log_file: ?std.fs.File,
    jsonl_file: ?std.fs.File,
    metrics_buffer: std.ArrayList(StepMetrics),
    allocator: std.mem.Allocator,
    max_buffer_size: usize,

    const Self = @This();

    pub fn init(
        allocator: std.mem.Allocator,
        config: TelemetryConfig,
        rank: usize,
    ) !Self {
        var log_file: ?std.fs.File = null;
        var jsonl_file: ?std.fs.File = null;

        if (rank == 0) {
            log_file = try std.fs.cwd().createFile(config.log_file, .{ .truncate = false });
            if (log_file) |f| {
                try f.seekFromEnd(0);
            }

            const jsonl_path = try std.fmt.allocPrint(allocator, "{s}.jsonl", .{config.log_file});
            defer allocator.free(jsonl_path);
            jsonl_file = try std.fs.cwd().createFile(jsonl_path, .{ .truncate = false });
            if (jsonl_file) |f| {
                try f.seekFromEnd(0);
            }
        }

        return .{
            .config = config,
            .rank = rank,
            .log_file = log_file,
            .jsonl_file = jsonl_file,
            .metrics_buffer = std.ArrayList(StepMetrics).init(allocator),
            .allocator = allocator,
            .max_buffer_size = 10000,
        };
    }

    pub fn deinit(self: *Self) void {
        if (self.log_file) |f| f.close();
        if (self.jsonl_file) |f| f.close();
        self.metrics_buffer.deinit();
    }

    pub fn logStep(self: *Self, metrics: StepMetrics) !void {
        if (self.rank != 0) return;

        if (self.metrics_buffer.items.len >= self.max_buffer_size) {
            try self.metrics_buffer.replaceRange(0, 1, &[_]StepMetrics{});
        }
        try self.metrics_buffer.append(metrics);

        if (self.jsonl_file) |f| {
            var buf_writer = std.io.bufferedWriter(f.writer());
            var writer = buf_writer.writer();

            try writer.print(
                "{{\"step\":{d},\"tokens\":{d},\"loss\":{d:.6e},\"lr\":{d:.6e},\"grad_norm\":{d:.4e},\"throughput\":{d:.2f},\"memory_used\":{d},\"memory_total\":{d},\"timestamp\":{d}}}\n",
                .{
                    metrics.step,
                    metrics.tokens,
                    metrics.loss,
                    metrics.lr,
                    metrics.grad_norm,
                    metrics.throughput,
                    metrics.memory_used,
                    metrics.memory_total,
                    metrics.timestamp,
                },
            );

            try buf_writer.flush();
        }
    }

    pub fn logMetrics(self: *Self, metrics: StepMetrics) !void {
        if (self.rank != 0) return;

        if (self.log_file) |f| {
            var buf_writer = std.io.bufferedWriter(f.writer());
            var writer = buf_writer.writer();

            try writer.print(
                "[{d}] step={d} loss={d:.4f} lr={d:.2e} tokens={d} throughput={d:.1f}\n",
                .{
                    std.time.timestamp(),
                    metrics.step,
                    metrics.loss,
                    metrics.lr,
                    metrics.tokens,
                    metrics.throughput,
                },
            );

            try buf_writer.flush();
        }
    }

    pub fn logHealthCheck(self: *Self, status: HealthStatus) !void {
        if (self.rank != 0) return;

        if (self.log_file) |f| {
            var buf_writer = std.io.bufferedWriter(f.writer());
            var writer = buf_writer.writer();
            try writer.print("[{d}] Health check: {}\n", .{ std.time.timestamp(), status });
            try buf_writer.flush();
        }
    }

    pub fn exportPrometheus(self: *Self, writer: anytype) !void {
        var last_step: usize = 0;
        var last_loss: f64 = 0.0;
        var last_throughput: f64 = 0.0;

        if (self.metrics_buffer.items.len > 0) {
            const last = self.metrics_buffer.items[self.metrics_buffer.items.len - 1];
            last_step = last.step;
            last_loss = @as(f64, last.loss);
            last_throughput = last.throughput;
        }

        try writer.writeAll("# HELP efla_training_step Current training step\n");
        try writer.writeAll("# TYPE efla_training_step gauge\n");
        try writer.print("efla_training_step {d}\n", .{last_step});

        try writer.writeAll("# HELP efla_training_loss Current training loss\n");
        try writer.writeAll("# TYPE efla_training_loss gauge\n");
        try writer.print("efla_training_loss {d:.6f}\n", .{last_loss});

        try writer.writeAll("# HELP efla_training_throughput Tokens per second\n");
        try writer.writeAll("# TYPE efla_training_throughput gauge\n");
        try writer.print("efla_training_throughput {d:.2f}\n", .{last_throughput});
    }
};

pub const HealthStatus = struct {
    gpu_healthy: bool,
    memory_healthy: bool,
    no_nan_inf: bool,
    temperature_ok: bool,

    pub fn isHealthy(self: HealthStatus) bool {
        return self.gpu_healthy and self.memory_healthy and
            self.no_nan_inf and self.temperature_ok;
    }

    pub fn format(
        self: HealthStatus,
        comptime fmt: []const u8,
        options: std.fmt.FormatOptions,
        writer: anytype,
    ) !void {
        _ = fmt;
        _ = options;
        try writer.print(
            "HealthStatus{{ gpu={}, memory={}, nan_inf={}, temp={} }}",
            .{
                self.gpu_healthy,
                self.memory_healthy,
                self.no_nan_inf,
                self.temperature_ok,
            },
        );
    }
};

pub fn checkNaNInf(ptr: [*]const f32, len: usize) bool {
    for (0..len) |i| {
        if (std.math.isNan(ptr[i]) or std.math.isInf(ptr[i])) {
            return true;
        }
    }
    return false;
}

pub fn logFn(
    comptime message_level: std.log.Level,
    comptime scope: @TypeOf(.enum_literal),
    comptime format: []const u8,
    args: anytype,
) void {
    const level_txt = switch (message_level) {
        .err => "ERROR",
        .warn => "WARN",
        .info => "INFO",
        .debug => "DEBUG",
    };

    const scope_txt = if (scope == .default) "" else @tagName(scope);

    const stderr = std.io.getStdErr().writer();

    const timestamp = std.time.timestamp();

    if (scope_txt.len > 0) {
        stderr.print("[{d}] [{s}] {s}: ", .{ timestamp, level_txt, scope_txt }) catch {};
    } else {
        stderr.print("[{d}] [{s}] ", .{ timestamp, level_txt }) catch {};
    }
    stderr.print(format ++ "\n", args) catch {};
}

pub const MemoryStats = struct {
    allocated: usize,
    freed: usize,
    current: usize,
    peak: usize,

    pub fn init() MemoryStats {
        return .{
            .allocated = 0,
            .freed = 0,
            .current = 0,
            .peak = 0,
        };
    }

    pub fn allocate(self: *MemoryStats, size: usize) void {
        self.allocated += size;
        self.current += size;
        if (self.current > self.peak) {
            self.peak = self.current;
        }
    }

    pub fn free(self: *MemoryStats, size: usize) void {
        if (size > self.current) {
            self.current = 0;
        } else {
            self.current -= size;
        }
        self.freed += size;
    }
};

pub const TrainingStats = struct {
    total_steps: usize,
    total_tokens: usize,
    total_time_us: i64,
    losses: std.ArrayList(f32),
    learning_rates: std.ArrayList(f32),
    grad_norms: std.ArrayList(f32),
    start_time_us: i64,
    allocator: std.mem.Allocator,

    pub fn init(allocator: std.mem.Allocator) TrainingStats {
        return .{
            .total_steps = 0,
            .total_tokens = 0,
            .total_time_us = 0,
            .losses = std.ArrayList(f32).init(allocator),
            .learning_rates = std.ArrayList(f32).init(allocator),
            .grad_norms = std.ArrayList(f32).init(allocator),
            .start_time_us = std.time.microTimestamp(),
            .allocator = allocator,
        };
    }

    pub fn deinit(self: *TrainingStats) void {
        self.losses.deinit();
        self.learning_rates.deinit();
        self.grad_norms.deinit();
    }

    pub fn record(self: *TrainingStats, loss: f32, lr: f32, grad_norm: f32, tokens: usize) !void {
        self.total_steps += 1;
        self.total_tokens += tokens;
        self.total_time_us = std.time.microTimestamp() - self.start_time_us;
        try self.losses.append(loss);
        try self.learning_rates.append(lr);
        try self.grad_norms.append(grad_norm);
    }

    pub fn avgLoss(self: *TrainingStats, window: usize) f32 {
        if (self.losses.items.len == 0) return 0.0;

        const actual_window = if (window == 0) self.losses.items.len else window;
        const start = if (self.losses.items.len > actual_window)
            self.losses.items.len - actual_window
        else
            0;

        var sum: f32 = 0.0;
        for (self.losses.items[start..]) |l| {
            sum += l;
        }

        return sum / @as(f32, @floatFromInt(self.losses.items.len - start));
    }

    pub fn throughput(self: *TrainingStats) f64 {
        const elapsed_us = std.time.microTimestamp() - self.start_time_us;
        if (elapsed_us <= 0) return 0.0;
        return (@as(f64, @floatFromInt(self.total_tokens)) * 1_000_000.0) / @as(f64, @floatFromInt(elapsed_us));
    }
};

const std = @import("std");

/// Error types for the training system
pub const TrainingError = error{
    // Configuration errors
    InvalidConfig,
    MissingConfig,
    MissingConfigPath,
    InvalidParameter,

    // Model errors
    InvalidHiddenDim,
    InvalidNumLayers,
    InvalidNumHeads,
    InvalidHeadDim,
    HiddenDimNotDivisibleByHeads,

    // Training errors
    InvalidBatchSize,
    InvalidGradientAccumulation,
    NaN,
    Inf,
    GradientOverflow,
    LossOverflow,

    // Runtime errors
    NoGpuAvailable,
    NotEnoughGpus,
    InvalidWorldSize,
    InvalidRank,

    // Data errors
    MissingDataPath,
    MissingData,
    InvalidData,
    IndexOutOfRange,

    // Checkpoint errors
    MissingCheckpoint,
    MissingCheckpointPath,
    InvalidCheckpoint,
    InvalidCheckpointFile,
    CheckpointCorrupted,

    // Tokenizer errors
    MissingCorpus,
    MissingTokenizer,
    InvalidToken,
    InvalidTokenizerFile,

    // IO errors
    FileNotFound,
    IoError,
};

/// Result type for operations that can fail
pub fn Result(comptime T: type) type {
    return union(enum) {
        ok: T,
        err: TrainingError,

        pub fn isOk(self: @This()) bool {
            return switch (self) {
                .ok => true,
                .err => false,
            };
        }

        pub fn isErr(self: @This()) bool {
            return !self.isOk();
        }

        pub fn unwrap(self: @This()) T {
            return switch (self) {
                .ok => |v| v,
                .err => |e| std.debug.panic("unwrap on error: {}", .{e}),
            };
        }

        pub fn unwrapOr(self: @This(), default: T) T {
            return switch (self) {
                .ok => |v| v,
                .err => default,
            };
        }
    };
}

/// Error context with additional information
pub const ErrorContext = struct {
    err: TrainingError,
    message: []const u8,
    file: []const u8,
    line: usize,

    pub fn format(
        self: ErrorContext,
        comptime fmt: []const u8,
        options: std.fmt.FormatOptions,
        writer: anytype,
    ) !void {
        _ = fmt;
        _ = options;
        try writer.print("{s}:{d}: {s}: {s}", .{ self.file, self.line, @errorName(self.err), self.message });
    }
};

/// Create an error with context
pub fn errWithContext(err: TrainingError, comptime message: []const u8) ErrorContext {
    return .{
        .err = err,
        .message = message,
        .file = @src().file,
        .line = @src().line,
    };
}

/// Check for NaN or Inf
pub fn checkFinite(value: f32) TrainingError!void {
    if (std.math.isNan(value)) return TrainingError.NaN;
    if (std.math.isInf(value)) return TrainingError.Inf;
}

/// Check slice for NaN or Inf
pub fn checkFiniteSlice(values: []const f32) TrainingError!void {
    for (values) |v| {
        try checkFinite(v);
    }
}

test "checkFinite" {
    try checkFinite(1.0);
    try checkFinite(-1.0);
    try checkFinite(0.0);

    try std.testing.expectError(TrainingError.NaN, checkFinite(std.math.nan(f32)));
    try std.testing.expectError(TrainingError.Inf, checkFinite(std.math.inf(f32)));
}

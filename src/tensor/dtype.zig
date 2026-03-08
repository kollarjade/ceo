const std = @import("std");

/// Numeric data types supported by the training system
pub const DType = enum(u8) {
    fp32 = 0,
    fp16 = 1,
    bf16 = 2,
    fp8_e4m3 = 3,
    fp8_e5m2 = 4,
    fp4 = 5,
    int8 = 6,
    int32 = 7,
    int64 = 8,
    bool_ = 9,

    pub fn sizeBytes(self: DType) usize {
        return switch (self) {
            .fp32 => 4,
            .fp16 => 2,
            .bf16 => 2,
            .fp8_e4m3 => 1,
            .fp8_e5m2 => 1,
            .fp4 => 1, // Packed as 2 per byte
            .int8 => 1,
            .int32 => 4,
            .int64 => 8,
            .bool_ => 1,
        };
    }

    pub fn isFloatingPoint(self: DType) bool {
        return switch (self) {
            .fp32, .fp16, .bf16, .fp8_e4m3, .fp8_e5m2, .fp4 => true,
            else => false,
        };
    }

    pub fn isQuantized(self: DType) bool {
        return switch (self) {
            .fp8_e4m3, .fp8_e5m2, .fp4 => true,
            else => false,
        };
    }

    pub fn accumulationDType(self: DType) DType {
        return switch (self) {
            .fp32 => .fp32,
            .fp16, .bf16 => .fp32,
            .fp8_e4m3, .fp8_e5m2 => .bf16,
            .fp4 => .bf16,
            else => .fp32,
        };
    }

    pub fn min(self: DType) f64 {
        return switch (self) {
            .fp32 => -std.math.floatMax(f32),
            .fp16 => -65504.0,
            .bf16 => -std.math.floatMax(f32), // Same range as fp32
            .fp8_e4m3 => -448.0,
            .fp8_e5m2 => -57344.0,
            .fp4 => -6.0,
            else => 0.0,
        };
    }

    pub fn max(self: DType) f64 {
        return switch (self) {
            .fp32 => std.math.floatMax(f32),
            .fp16 => 65504.0,
            .bf16 => std.math.floatMax(f32),
            .fp8_e4m3 => 448.0,
            .fp8_e5m2 => 57344.0,
            .fp4 => 6.0,
            else => 0.0,
        };
    }
};

/// Scale factor for quantized tensors
pub const ScaleFactor = extern struct {
    scale: f32,
    inv_scale: f32,
    amax: f32,
    amax_history: [16]f32,

    pub fn init() ScaleFactor {
        return .{
            .scale = 1.0,
            .inv_scale = 1.0,
            .amax = 0.0,
            .amax_history = [_]f32{0.0} ** 16,
        };
    }

    pub fn fromScale(scale: f32) ScaleFactor {
        return .{
            .scale = scale,
            .inv_scale = if (scale != 0.0) 1.0 / scale else 0.0,
            .amax = 0.0,
            .amax_history = [_]f32{0.0} ** 16,
        };
    }

    pub fn update(self: *ScaleFactor, new_amax: f32) void {
        // Shift history
        var i: usize = 15;
        while (i > 0) : (i -= 1) {
            self.amax_history[i] = self.amax_history[i - 1];
        }
        self.amax_history[0] = new_amax;
        self.amax = new_amax;

        // Use max of recent history for stability
        var max_amax: f32 = 0.0;
        for (self.amax_history) |a| {
            if (a > max_amax) max_amax = a;
        }

        if (max_amax > 0.0) {
            self.scale = 448.0 / max_amax; // FP8 E4M3 max
            self.inv_scale = max_amax / 448.0;
        }
    }
};

/// Block-wise scaling for fine-grained quantization
pub const BlockScale = extern struct {
    block_size: u32,
    num_blocks: u32,
    scales: [*]ScaleFactor,

    pub fn init(allocator: std.mem.Allocator, total_elements: usize, block_size: usize) !BlockScale {
        const num_blocks = (total_elements + block_size - 1) / block_size;
        const scales = try allocator.alloc(ScaleFactor, num_blocks);

        for (scales) |*s| {
            s.* = ScaleFactor.init();
        }

        return .{
            .block_size = @intCast(block_size),
            .num_blocks = @intCast(num_blocks),
            .scales = scales.ptr,
        };
    }

    pub fn deinit(self: *BlockScale, allocator: std.mem.Allocator) void {
        allocator.free(self.scales[0..self.num_blocks]);
    }

    pub fn getScale(self: BlockScale, block_idx: usize) *ScaleFactor {
        return &self.scales[block_idx];
    }
};

/// FP8 E4M3 representation
pub const FP8_E4M3 = packed struct(u8) {
    mantissa: u3,
    exponent: u4,
    sign: u1,

    pub const MAX_VALUE: f32 = 448.0;
    pub const MIN_POSITIVE: f32 = 0.015625; // 2^-6

    pub fn fromFloat32(val: f32) FP8_E4M3 {
        const bits: u32 = @bitCast(val);
        const sign: u1 = @intCast(bits >> 31);

        const abs_val = @abs(val);
        if (abs_val > MAX_VALUE) {
            // Saturate to max
            return .{ .sign = sign, .exponent = 0b1000, .mantissa = 0b111 };
        }
        if (abs_val < MIN_POSITIVE and abs_val > 0.0) {
            // Flush to zero
            return .{ .sign = 0, .exponent = 0, .mantissa = 0 };
        }

        // Convert to E4M3 format
        const exp_bits = bits & 0x7F800000;
        const mant_bits = bits & 0x007FFFFF;

        const fp32_exp: i32 = @intCast(exp_bits >> 23);
        const fp32_exp_biased = fp32_exp - 127;

        // E4M3 has bias of 7
        const new_exp = fp32_exp_biased + 7;
        if (new_exp <= 0) {
            // Subnormal or zero
            return .{ .sign = sign, .exponent = 0, .mantissa = 0 };
        }
        if (new_exp >= 15) {
            // Saturate
            return .{ .sign = sign, .exponent = 0b1000, .mantissa = 0b111 };
        }

        const new_mant: u3 = @truncate(mant_bits >> 20);
        return .{ .sign = sign, .exponent = @intCast(new_exp), .mantissa = new_mant };
    }

    pub fn toFloat32(self: FP8_E4M3) f32 {
        if (self.exponent == 0 and self.mantissa == 0) {
            return if (self.sign == 1) -0.0 else 0.0;
        }

        // E4M3 bias is 7, FP32 bias is 127
        const fp32_exp: i32 = @as(i32, self.exponent) - 7 + 127;
        const fp32_mant: u32 = @as(u32, self.mantissa) << 20;

        const bits: u32 = (@as(u32, self.sign) << 31) | (@as(u32, @intCast(fp32_exp)) << 23) | fp32_mant;
        return @bitCast(bits);
    }
};

/// FP8 E5M2 representation
pub const FP8_E5M2 = packed struct(u8) {
    mantissa: u2,
    exponent: u5,
    sign: u1,

    pub const MAX_VALUE: f32 = 57344.0;
    pub const MIN_POSITIVE: f32 = 0.00006103515625; // 2^-14

    pub fn fromFloat32(val: f32) FP8_E5M2 {
        const bits: u32 = @bitCast(val);
        const sign: u1 = @intCast(bits >> 31);

        const abs_val = @abs(val);
        if (std.math.isInf(abs_val)) {
            return .{ .sign = sign, .exponent = 0b11111, .mantissa = 0 };
        }
        if (std.math.isNan(abs_val)) {
            return .{ .sign = sign, .exponent = 0b11111, .mantissa = 1 };
        }
        if (abs_val > MAX_VALUE) {
            // Overflow to infinity
            return .{ .sign = sign, .exponent = 0b11111, .mantissa = 0 };
        }
        if (abs_val < MIN_POSITIVE and abs_val > 0.0) {
            // Flush to zero
            return .{ .sign = 0, .exponent = 0, .mantissa = 0 };
        }

        const exp_bits = bits & 0x7F800000;
        const mant_bits = bits & 0x007FFFFF;

        const fp32_exp: i32 = @intCast(exp_bits >> 23);
        const fp32_exp_biased = fp32_exp - 127;

        // E5M2 has bias of 15
        const new_exp = fp32_exp_biased + 15;
        if (new_exp <= 0) {
            return .{ .sign = sign, .exponent = 0, .mantissa = 0 };
        }
        if (new_exp >= 31) {
            return .{ .sign = sign, .exponent = 0b11111, .mantissa = 0 };
        }

        const new_mant: u2 = @truncate(mant_bits >> 21);
        return .{ .sign = sign, .exponent = @intCast(new_exp), .mantissa = new_mant };
    }

    pub fn toFloat32(self: FP8_E5M2) f32 {
        if (self.exponent == 0 and self.mantissa == 0) {
            return if (self.sign == 1) -0.0 else 0.0;
        }
        if (self.exponent == 0b11111) {
            if (self.mantissa == 0) {
                return if (self.sign == 1) std.math.inf(f32) * -1.0 else std.math.inf(f32);
            } else {
                return std.math.nan(f32);
            }
        }

        const fp32_exp: i32 = @as(i32, self.exponent) - 15 + 127;
        const fp32_mant: u32 = @as(u32, self.mantissa) << 21;

        const bits: u32 = (@as(u32, self.sign) << 31) | (@as(u32, @intCast(fp32_exp)) << 23) | fp32_mant;
        return @bitCast(bits);
    }
};

/// BF16 representation
pub const BF16 = packed struct(u16) {
    mantissa: u7,
    exponent: u8,
    sign: u1,

    pub fn fromFloat32(val: f32) BF16 {
        const bits: u32 = @bitCast(val);
        // BF16 is just the upper 16 bits of FP32
        return @bitCast(@as(u16, @truncate(bits >> 16)));
    }

    pub fn toFloat32(self: BF16) f32 {
        const bits: u32 = @as(u32, @as(u16, @bitCast(self))) << 16;
        return @bitCast(bits);
    }
};

/// FP16 representation
pub const FP16 = packed struct(u16) {
    mantissa: u10,
    exponent: u5,
    sign: u1,

    pub fn fromFloat32(val: f32) FP16 {
        const bits: u32 = @bitCast(val);
        const sign: u1 = @intCast(bits >> 31);

        const abs_val = @abs(val);
        if (abs_val > 65504.0) {
            // Overflow
            return .{ .sign = sign, .exponent = 0b11111, .mantissa = 0 };
        }
        if (abs_val < 6.10352e-5 and abs_val > 0.0) {
            // Subnormal handling
            const scale: f32 = 0x1.0p24;
            const scaled = abs_val * scale;
            const scaled_bits: u32 = @bitCast(scaled);
            const mant: u10 = @truncate((scaled_bits & 0x007FFFFF) >> 13);
            return .{ .sign = sign, .exponent = 0, .mantissa = mant };
        }

        // Normal number
        const exp_bits: i32 = @intCast((bits & 0x7F800000) >> 23);
        const fp16_exp = exp_bits - 127 + 15;

        if (fp16_exp <= 0) {
            return .{ .sign = sign, .exponent = 0, .mantissa = 0 };
        }
        if (fp16_exp >= 31) {
            return .{ .sign = sign, .exponent = 0b11111, .mantissa = 0 };
        }

        const mant: u10 = @truncate((bits & 0x007FFFFF) >> 13);
        return .{ .sign = sign, .exponent = @intCast(fp16_exp), .mantissa = mant };
    }

    pub fn toFloat32(self: FP16) f32 {
        if (self.exponent == 0) {
            if (self.mantissa == 0) {
                return if (self.sign == 1) -0.0 else 0.0;
            }
            // Subnormal
            const scale: f32 = 0x1.0p-24;
            const val = @as(f32, @floatFromInt(self.mantissa)) * scale;
            return if (self.sign == 1) -val else val;
        }
        if (self.exponent == 0b11111) {
            if (self.mantissa == 0) {
                return if (self.sign == 1) std.math.inf(f32) * -1.0 else std.math.inf(f32);
            } else {
                return std.math.nan(f32);
            }
        }

        const fp32_exp: i32 = @as(i32, self.exponent) - 15 + 127;
        const fp32_mant: u32 = @as(u32, self.mantissa) << 13;

        const bits: u32 = (@as(u32, self.sign) << 31) | (@as(u32, @intCast(fp32_exp)) << 23) | fp32_mant;
        return @bitCast(bits);
    }
};

/// Conversion utilities
pub fn castSlice(comptime from: DType, comptime to: DType, input: []const u8, output: []u8) void {
    const from_size = from.sizeBytes();
    const to_size = to.sizeBytes();
    const n = input.len / from_size;

    switch (from) {
        .fp32 => {
            const in_slice = std.mem.bytesAsSlice(f32, input);
            switch (to) {
                .fp16 => {
                    const out_slice = std.mem.bytesAsSlice(FP16, output);
                    for (in_slice, out_slice) |v, *o| o.* = FP16.fromFloat32(v);
                },
                .bf16 => {
                    const out_slice = std.mem.bytesAsSlice(BF16, output);
                    for (in_slice, out_slice) |v, *o| o.* = BF16.fromFloat32(v);
                },
                .fp8_e4m3 => {
                    const out_slice = std.mem.bytesAsSlice(FP8_E4M3, output);
                    for (in_slice, out_slice) |v, *o| o.* = FP8_E4M3.fromFloat32(v);
                },
                .fp8_e5m2 => {
                    const out_slice = std.mem.bytesAsSlice(FP8_E5M2, output);
                    for (in_slice, out_slice) |v, *o| o.* = FP8_E5M2.fromFloat32(v);
                },
                else => @compileError("Unsupported cast"),
            }
        },
        .fp16 => {
            const in_slice = std.mem.bytesAsSlice(FP16, input);
            switch (to) {
                .fp32 => {
                    const out_slice = std.mem.bytesAsSlice(f32, output);
                    for (in_slice, out_slice) |v, *o| o.* = v.toFloat32();
                },
                else => @compileError("Unsupported cast"),
            }
        },
        .bf16 => {
            const in_slice = std.mem.bytesAsSlice(BF16, input);
            switch (to) {
                .fp32 => {
                    const out_slice = std.mem.bytesAsSlice(f32, output);
                    for (in_slice, out_slice) |v, *o| o.* = v.toFloat32();
                },
                else => @compileError("Unsupported cast"),
            }
        },
        .fp8_e4m3 => {
            const in_slice = std.mem.bytesAsSlice(FP8_E4M3, input);
            switch (to) {
                .fp32 => {
                    const out_slice = std.mem.bytesAsSlice(f32, output);
                    for (in_slice, out_slice) |v, *o| o.* = v.toFloat32();
                },
                else => @compileError("Unsupported cast"),
            }
        },
        .fp8_e5m2 => {
            const in_slice = std.mem.bytesAsSlice(FP8_E5M2, input);
            switch (to) {
                .fp32 => {
                    const out_slice = std.mem.bytesAsSlice(f32, output);
                    for (in_slice, out_slice) |v, *o| o.* = v.toFloat32();
                },
                else => @compileError("Unsupported cast"),
            }
        },
        else => @compileError("Unsupported source dtype"),
    }
}

test "FP8_E4M3 roundtrip" {
    const test_vals = [_]f32{ 0.0, 1.0, -1.0, 448.0, -448.0, 0.5, 0.25, 100.0 };

    for (test_vals) |v| {
        const fp8 = FP8_E4M3.fromFloat32(v);
        const back = fp8.toFloat32();
        try std.testing.expectApproxEqRel(@abs(v), @abs(back), 0.1);
    }
}

test "FP16 roundtrip" {
    const test_vals = [_]f32{ 0.0, 1.0, -1.0, 65504.0, -65504.0, 0.5, 0.25, 100.0 };

    for (test_vals) |v| {
        const fp16 = FP16.fromFloat32(v);
        const back = fp16.toFloat32();
        try std.testing.expectApproxEqRel(@abs(v), @abs(back), 0.001);
    }
}

test "BF16 roundtrip" {
    const test_vals = [_]f32{ 0.0, 1.0, -1.0, 100.0, -100.0, 0.5, 0.25 };

    for (test_vals) |v| {
        const bf16 = BF16.fromFloat32(v);
        const back = bf16.toFloat32();
        try std.testing.expectApproxEqRel(@abs(v), @abs(back), 0.01);
    }
}
